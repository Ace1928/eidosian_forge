import pygame
import requests
from io import BytesIO

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
PALETTE_HEIGHT = 100
COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 165, 0),
    (128, 0, 128),
    (0, 255, 255),
]
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


# Load Cartoon Image
def load_image(url):
    response = requests.get(url)
    image = pygame.image.load(BytesIO(response.content))
    return image


# Example cartoon image URL (replace with a valid URL)
image_url = "https://example.com/cartoon_image.png"
cartoon_image = load_image(image_url)
cartoon_image = pygame.transform.scale(
    cartoon_image, (SCREEN_WIDTH, SCREEN_HEIGHT - PALETTE_HEIGHT)
)

# Screen Setup
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Color by Number Game")

# Palette Setup
palette_rects = []
for i, color in enumerate(COLORS):
    rect = pygame.Rect(
        i * (SCREEN_WIDTH // len(COLORS)),
        SCREEN_HEIGHT - PALETTE_HEIGHT,
        SCREEN_WIDTH // len(COLORS),
        PALETTE_HEIGHT,
    )
    palette_rects.append((rect, color))

# Game Variables
selected_color = BLACK
drawing = False

# Main Game Loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            if y >= SCREEN_HEIGHT - PALETTE_HEIGHT:
                for rect, color in palette_rects:
                    if rect.collidepoint(x, y):
                        selected_color = color
                        break
            else:
                drawing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
        elif event.type == pygame.MOUSEMOTION and drawing:
            x, y = event.pos
            if y < SCREEN_HEIGHT - PALETTE_HEIGHT:
                pygame.draw.circle(screen, selected_color, (x, y), 5)

    # Draw Everything
    screen.fill(WHITE)
    screen.blit(cartoon_image, (0, 0))
    for rect, color in palette_rects:
        pygame.draw.rect(screen, color, rect)
    pygame.display.flip()

pygame.quit()
